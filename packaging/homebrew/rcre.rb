class Rcre < Formula
  desc "Zero-dependency ISO C23 AI, Vision, LLM (GGUF/Q4_0/Q8_0), and High-Throughput I/O Engine"
  homepage "https://github.com/erolrecep/rcre"
  license "MIT"
  head "https://github.com/erolrecep/rcre.git", branch: "main"

  depends_on "cmake" => :build

  def install
    system "cmake", "-S", ".", "-B", "build", *std_cmake_args
    system "cmake", "--build", "build"
    system "cmake", "--install", "build"
  end

  test do
    (testpath/"probe.c").write <<~EOS
      #include <rcre.h>
      int main(void) {
        rcre_hal_init();
        return rcre_abi_version() >= 0x020000U ? 0 : 1;
      }
    EOS
    system ENV.cc, "-std=c2x", "probe.c", "-I#{include}", "-L#{lib}", "-lrcre", "-lm", "-lpthread", "-o", "probe"
    system "./probe"
  end
end
